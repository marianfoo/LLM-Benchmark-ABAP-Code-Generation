CLASS ltcl_test_do_algebra DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_do_algebra IMPLEMENTATION.

  METHOD test_case_1.
    DATA: operator TYPE TABLE OF string,
          operand  TYPE TABLE OF i,
          result   TYPE i.

    operator = VALUE #( ( `**` ) ( `*` ) ( `+` ) ).
    operand = VALUE #( ( 2 ) ( 3 ) ( 4 ) ( 5 ) ).

    result = z_humaneval_160=>do_algebra( operator = operator operand = operand ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 37
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: operator TYPE TABLE OF string,
          operand  TYPE TABLE OF i,
          result   TYPE i.

    operator = VALUE #( ( `+` ) ( `*` ) ( `-` ) ).
    operand = VALUE #( ( 2 ) ( 3 ) ( 4 ) ( 5 ) ).

    result = z_humaneval_160=>do_algebra( operator = operator operand = operand ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 9
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: operator TYPE TABLE OF string,
          operand  TYPE TABLE OF i,
          result   TYPE i.

    operator = VALUE #( ( `//` ) ( `*` ) ).
    operand = VALUE #( ( 7 ) ( 3 ) ( 4 ) ).

    result = z_humaneval_160=>do_algebra( operator = operator operand = operand ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = 8
    ).
  ENDMETHOD.

ENDCLASS.