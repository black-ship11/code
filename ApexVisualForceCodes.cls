1. go to developer console 
2. file -> new -> apex class -> Name as "CalculatorController"

Code -
public class CalculatorController {
    public Integer num1 { get; set; }
    public Integer num2 { get; set; }
    public String operation { get; set; }
    public String result { get; set; }

    public void calculate() {
        if (operation == 'Add') {
            result = 'Result: ' + (num1 + num2);
        } else if (operation == 'Subtract') {
            result = 'Result: ' + (num1 - num2);
        } else if (operation == 'Multiply') {
            result = 'Result: ' + (num1 * num2);
        } else if (operation == 'Divide') {
            result = 'Result: ' + ((Decimal)num1 / (Decimal)num2);
        } else {
            result = 'Invalid operation';
        }
    }
}

3. save 
4. go to setup 
5. search for visualforce pages
6. create new

code - 
<apex:page controller="CalculatorController">
    <h1>Simple Calculator</h1>

    <apex:form>
        <apex:pageBlock title="Calculator">
            <apex:pageBlockSection columns="2">

                <apex:inputText value="{!num1}" label="Number 1"/>
                <apex:inputText value="{!num2}" label="Number 2"/>

                <apex:selectList value="{!operation}" size="1" label="Operation">
                    <apex:selectOption itemLabel="Add" itemValue="Add"/>
                    <apex:selectOption itemLabel="Subtract" itemValue="Subtract"/>
                    <apex:selectOption itemLabel="Multiply" itemValue="Multiply"/>
                    <apex:selectOption itemLabel="Divide" itemValue="Divide"/>
                </apex:selectList>

                <apex:commandButton value="Calculate" action="{!calculate}" rerender="resultPanel"/>

            </apex:pageBlockSection>

            <apex:outputPanel id="resultPanel">
                <h2>{!result}</h2>
            </apex:outputPanel>
        </apex:pageBlock>
    </apex:form>
</apex:page>

7. name it anything
8 save and then preview... Done !