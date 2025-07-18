CLASS ltcl_test_z_humaneval_126 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_single_element FOR TESTING,
      test_sorted_ascending FOR TESTING,
      test_not_sorted FOR TESTING,
      test_sorted_six_elements FOR TESTING,
      test_sorted_seven_elements FOR TESTING,
      test_not_sorted_seven_elements FOR TESTING,
      test_empty_list FOR TESTING,
      test_single_one FOR TESTING,
      test_descending FOR TESTING,
      test_three_duplicates_first FOR TESTING,
      test_three_duplicates_second FOR TESTING,
      test_two_duplicates_allowed FOR TESTING,
      test_four_elements_sorted FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_126 IMPLEMENTATION.

  METHOD test_single_element.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 5 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_sorted_ascending.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_not_sorted.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 4 ) ( 5 ) ).
    cl_abap_unit_assert=>assert_false( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_sorted_six_elements.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ( 6 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_sorted_seven_elements.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ( 6 ) ( 7 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_not_sorted_seven_elements.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 4 ) ( 5 ) ( 6 ) ( 7 ) ).
    cl_abap_unit_assert=>assert_false( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_empty_list.
    DATA: lt_input TYPE int4_table.
    cl_abap_unit_assert=>assert_true( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_single_one.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_descending.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 3 ) ( 2 ) ( 1 ) ).
    cl_abap_unit_assert=>assert_false( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_three_duplicates_first.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ( 2 ) ( 2 ) ( 2 ) ( 3 ) ( 4 ) ).
    cl_abap_unit_assert=>assert_false( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_three_duplicates_second.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 3 ) ( 3 ) ( 4 ) ).
    cl_abap_unit_assert=>assert_false( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_two_duplicates_allowed.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ( 2 ) ( 2 ) ( 3 ) ( 3 ) ( 4 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

  METHOD test_four_elements_sorted.
    DATA: lt_input TYPE int4_table.
    lt_input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_126=>is_sorted( lt_input ) ).
  ENDMETHOD.

ENDCLASS.