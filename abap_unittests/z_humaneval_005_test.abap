CLASS ltcl_test_intersperse DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_empty_list FOR TESTING,
      test_multiple_elements FOR TESTING,
      test_repeated_elements FOR TESTING.

ENDCLASS.

CLASS ltcl_test_intersperse IMPLEMENTATION.

  METHOD test_empty_list.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    actual = z_humaneval_005=>intersperse( numbers = input delimeter = 7 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_multiple_elements.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 6 ) ( 3 ) ( 2 ) ).
    expected = VALUE #( ( 5 ) ( 8 ) ( 6 ) ( 8 ) ( 3 ) ( 8 ) ( 2 ) ).

    actual = z_humaneval_005=>intersperse( numbers = input delimeter = 8 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_repeated_elements.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 2 ) ( 2 ) ( 2 ) ).
    expected = VALUE #( ( 2 ) ( 2 ) ( 2 ) ( 2 ) ( 2 ) ).

    actual = z_humaneval_005=>intersperse( numbers = input delimeter = 2 ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

ENDCLASS.