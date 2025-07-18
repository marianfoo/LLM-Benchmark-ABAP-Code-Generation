CLASS ltcl_test_filter_integers DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    TYPES: BEGIN OF ty_any_value,
             value TYPE string,
           END OF ty_any_value.
    TYPES: tt_any_values TYPE TABLE OF ty_any_value.

    METHODS: test_empty_list FOR TESTING,
      test_mixed_values FOR TESTING,
      test_multiple_integers FOR TESTING.
ENDCLASS.

CLASS ltcl_test_filter_integers IMPLEMENTATION.
  METHOD test_empty_list.
    DATA: lt_input    TYPE tt_any_values,
          lt_expected TYPE int4_table,
          lt_result   TYPE int4_table.

    lt_result = z_humaneval_022=>filter_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_result
      exp = lt_expected
    ).
  ENDMETHOD.

  METHOD test_mixed_values.
    DATA: lt_input    TYPE tt_any_values,
          lt_expected TYPE int4_table,
          lt_result   TYPE int4_table.

    lt_input = VALUE #( ( value = '4' )
                       ( value = '{}' )
                       ( value = '[]' )
                       ( value = '23.2' )
                       ( value = '9' )
                       ( value = 'adasd' ) ).

    lt_expected = VALUE #( ( 4 ) ( 9 ) ).

    lt_result = z_humaneval_022=>filter_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_result
      exp = lt_expected
    ).
  ENDMETHOD.

  METHOD test_multiple_integers.
    DATA: lt_input    TYPE tt_any_values,
          lt_expected TYPE int4_table,
          lt_result   TYPE int4_table.

    lt_input = VALUE #( ( value = '3' )
                       ( value = 'c' )
                       ( value = '3' )
                       ( value = '3' )
                       ( value = 'a' )
                       ( value = 'b' ) ).

    lt_expected = VALUE #( ( 3 ) ( 3 ) ( 3 ) ).

    lt_result = z_humaneval_022=>filter_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_result
      exp = lt_expected
    ).
  ENDMETHOD.
ENDCLASS.