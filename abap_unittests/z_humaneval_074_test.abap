CLASS ltcl_test_total_match DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_empty_lists FOR TESTING,
      test_hi_admin_vs_hi_hi FOR TESTING,
      test_hi_admin_vs_long_list FOR TESTING,
      test_four_vs_numbers FOR TESTING,
      test_hi_admin_vs_hi_hi_case FOR TESTING,
      test_hi_admin_vs_hi_hi_hi FOR TESTING,
      test_hi_admin_vs_hi_hii FOR TESTING,
      test_empty_vs_this FOR TESTING,
      test_this_vs_empty FOR TESTING.
ENDCLASS.

CLASS ltcl_test_total_match IMPLEMENTATION.
  METHOD test_empty_lists.
    DATA: lst1     TYPE string_table,
          lst2     TYPE string_table,
          result   TYPE string_table,
          expected TYPE string_table.

    result = z_humaneval_074=>total_match( lst1 = lst1 lst2 = lst2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_hi_admin_vs_hi_hi.
    DATA: lst1     TYPE string_table,
          lst2     TYPE string_table,
          result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'hi' TO lst1.
    APPEND 'admin' TO lst1.
    APPEND 'hi' TO lst2.
    APPEND 'hi' TO lst2.
    APPEND 'hi' TO expected.
    APPEND 'hi' TO expected.

    result = z_humaneval_074=>total_match( lst1 = lst1 lst2 = lst2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_hi_admin_vs_long_list.
    DATA: lst1     TYPE string_table,
          lst2     TYPE string_table,
          result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'hi' TO lst1.
    APPEND 'admin' TO lst1.
    APPEND 'hi' TO lst2.
    APPEND 'hi' TO lst2.
    APPEND 'admin' TO lst2.
    APPEND 'project' TO lst2.
    APPEND 'hi' TO expected.
    APPEND 'admin' TO expected.

    result = z_humaneval_074=>total_match( lst1 = lst1 lst2 = lst2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_four_vs_numbers.
    DATA: lst1     TYPE string_table,
          lst2     TYPE string_table,
          result   TYPE string_table,
          expected TYPE string_table.

    APPEND '4' TO lst1.
    APPEND '1' TO lst2.
    APPEND '2' TO lst2.
    APPEND '3' TO lst2.
    APPEND '4' TO lst2.
    APPEND '5' TO lst2.
    APPEND '4' TO expected.

    result = z_humaneval_074=>total_match( lst1 = lst1 lst2 = lst2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_hi_admin_vs_hi_hi_case.
    DATA: lst1     TYPE string_table,
          lst2     TYPE string_table,
          result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'hi' TO lst1.
    APPEND 'admin' TO lst1.
    APPEND 'hI' TO lst2.
    APPEND 'Hi' TO lst2.
    APPEND 'hI' TO expected.
    APPEND 'Hi' TO expected.

    result = z_humaneval_074=>total_match( lst1 = lst1 lst2 = lst2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_hi_admin_vs_hi_hi_hi.
    DATA: lst1     TYPE string_table,
          lst2     TYPE string_table,
          result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'hi' TO lst1.
    APPEND 'admin' TO lst1.
    APPEND 'hI' TO lst2.
    APPEND 'hi' TO lst2.
    APPEND 'hi' TO lst2.
    APPEND 'hI' TO expected.
    APPEND 'hi' TO expected.
    APPEND 'hi' TO expected.

    result = z_humaneval_074=>total_match( lst1 = lst1 lst2 = lst2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_hi_admin_vs_hi_hii.
    DATA: lst1     TYPE string_table,
          lst2     TYPE string_table,
          result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'hi' TO lst1.
    APPEND 'admin' TO lst1.
    APPEND 'hI' TO lst2.
    APPEND 'hi' TO lst2.
    APPEND 'hii' TO lst2.
    APPEND 'hi' TO expected.
    APPEND 'admin' TO expected.

    result = z_humaneval_074=>total_match( lst1 = lst1 lst2 = lst2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_empty_vs_this.
    DATA: lst1     TYPE string_table,
          lst2     TYPE string_table,
          result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'this' TO lst2.

    result = z_humaneval_074=>total_match( lst1 = lst1 lst2 = lst2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_this_vs_empty.
    DATA: lst1     TYPE string_table,
          lst2     TYPE string_table,
          result   TYPE string_table,
          expected TYPE string_table.

    APPEND 'this' TO lst1.

    result = z_humaneval_074=>total_match( lst1 = lst1 lst2 = lst2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.
ENDCLASS.