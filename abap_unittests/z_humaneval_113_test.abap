CLASS ltcl_test_odd_count DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: string_table TYPE STANDARD TABLE OF string WITH DEFAULT KEY.

    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_odd_count IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    APPEND '1234567' TO input.
    APPEND 'the number of odd elements 4n the str4ng 4 of the 4nput.' TO expected.

    actual = z_humaneval_113=>odd_count( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test 1 failed'
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    APPEND '3' TO input.
    APPEND '11111111' TO input.
    APPEND 'the number of odd elements 1n the str1ng 1 of the 1nput.' TO expected.
    APPEND 'the number of odd elements 8n the str8ng 8 of the 8nput.' TO expected.

    actual = z_humaneval_113=>odd_count( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test 2 failed'
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    APPEND '271' TO input.
    APPEND '137' TO input.
    APPEND '314' TO input.
    APPEND 'the number of odd elements 2n the str2ng 2 of the 2nput.' TO expected.
    APPEND 'the number of odd elements 3n the str3ng 3 of the 3nput.' TO expected.
    APPEND 'the number of odd elements 2n the str2ng 2 of the 2nput.' TO expected.

    actual = z_humaneval_113=>odd_count( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test 3 failed'
    ).
  ENDMETHOD.

ENDCLASS.